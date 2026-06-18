(define (domain region-object-rearrangement-explicit-state-domain)
    (:requirements :typing :adl :action-costs)
    (:types
        place dsg_object region - object
    )

    (:predicates
        (object-in-place ?o - dsg_object ?p - place)
        (place-in-region ?p - place ?r - region)
        (connected ?s - place ?t - place)

        (at-place ?p - place)
        (at-object ?o - dsg_object)
        (in-region ?r - region)

        (visited-place ?p - place)
        (visited-object ?o - dsg_object)
        (visited-region ?r - region)

        (hand-full)
        (holding ?o - dsg_object)

        (suspicious ?o - dsg_object)
        (unsafe-place ?p - place)
    )

    (:functions
        (distance ?s - place ?t - place)
        (total-cost)
    )

    (:action goto-poi
        :parameters (?s - place ?t - place)
        :precondition
            (and
                (at-place ?s)
                (not (unsafe-place ?t))
                (or
                    (connected ?s ?t)
                    (connected ?t ?s)
                )
            )
        :effect
            (and
                (not (at-place ?s))
                (at-place ?t)
                (visited-place ?t)
                (forall (?r - region)
                    (when (place-in-region ?s ?r)
                        (not (in-region ?r))
                    )
                )
                (forall (?r - region)
                    (when (place-in-region ?t ?r)
                        (and
                            (in-region ?r)
                            (visited-region ?r)
                        )
                    )
                )
                (forall (?o - dsg_object)
                    (when (object-in-place ?o ?s)
                        (not (at-object ?o))
                    )
                )
                (forall (?o - dsg_object)
                    (when (object-in-place ?o ?t)
                        (and
                            (at-object ?o)
                            (visited-object ?o)
                        )
                    )
                )
                (increase (total-cost) (distance ?s ?t))
            )
    )

    (:action pick-object
        :parameters (?o - dsg_object ?p - place)
        :precondition
            (and
                (not (hand-full))
                (not (suspicious ?o))
                (at-place ?p)
                (object-in-place ?o ?p)
            )
        :effect
            (and
                (holding ?o)
                (hand-full)
                (not (object-in-place ?o ?p))
                (increase (total-cost) 10)
            )
    )

    (:action place-object
        :parameters (?o - dsg_object ?p - place)
        :precondition
            (and
                (holding ?o)
                (at-place ?p)
            )
        :effect
            (and
                (not (holding ?o))
                (not (hand-full))
                (object-in-place ?o ?p)
                (increase (total-cost) 10)
            )
    )

    (:action inspect
        :parameters (?o - dsg_object ?p - place ?t - place)
        :precondition
            (and
                (suspicious ?o)
                (object-in-place ?o ?p)
                (at-place ?t)
                (or
                    (connected ?p ?t)
                    (connected ?t ?p)
                )
            )
        :effect
            (and
                (not (suspicious ?o))
                (not (unsafe-place ?p))
                (increase (total-cost) 10)
            )
    )

)
